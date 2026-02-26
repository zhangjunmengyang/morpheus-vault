---
brief: "PEFT 方法对比——LoRA/Prefix Tuning/Prompt Tuning/Adapter 四大流派的原理、显存占用、性能和适用场景完整对比表；面试被问参数高效微调选型时的标准参考，直接给出决策矩阵。"
title: "PEFT 方法对比：LoRA / Prefix / Prompt / Adapter"
date: 2026-02-14
domain: AI/LLM/SFT
tags: [LLM, PEFT, LoRA, Fine-tuning, Efficiency, Parameter-Efficient]
rating: 4
status: active
---

# PEFT 方法对比

参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）是在不修改大部分预训练参数的情况下，通过少量可训练参数实现模型适配的技术。本文对比主流 PEFT 方法的原理、实现和应用场景。

## LoRA 原理与实现

### 核心思想
LoRA（Low-Rank Adaptation）基于低秩分解的思想，认为微调过程中的权重变化矩阵具有低秩特性：

$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # 原始前向传播
        result = F.linear(x, self.weight)
        
        # LoRA 分支
        lora_result = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        
        return result + lora_result * self.scale

# 实际应用示例
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False  # 冻结原始权重
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
    
    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
```

### LoRA 训练流程
```python
def convert_to_lora(model, target_modules, rank=16):
    """将指定模块转换为 LoRA"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent = model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                
                lora_layer = LoRALinear(module, rank=rank)
                setattr(parent, name.split('.')[-1], lora_layer)
    
    return model

# 使用示例
model = AutoModel.from_pretrained("llama-7b")
model = convert_to_lora(model, target_modules=["q_proj", "v_proj"], rank=16)

# 只训练 LoRA 参数
lora_params = [p for n, p in model.named_parameters() if "lora" in n]
optimizer = AdamW(lora_params, lr=1e-4)
```

## QLoRA：量化 + LoRA

### 原理
QLoRA 结合 4-bit 量化和 LoRA，实现极致的内存效率：

```python
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

# QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 量化
    bnb_4bit_quant_type="nf4",             # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16,   # 计算数据类型
    bnb_4bit_use_double_quant=True,        # 双重量化
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 应用 LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,                    # rank
    lora_alpha=128,         # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### 内存优化技巧
```python
class QLoRAOptimizedTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # 启用梯度检查点
        self.model.gradient_checkpointing_enable()
        
        # 优化器状态卸载
        self.optimizer = bnb.optim.PagedAdamW32bit(
            self.model.parameters(), 
            lr=1e-4
        )
    
    def train_step(self, batch):
        # 混合精度训练
        with torch.cuda.amp.autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # 梯度缩放
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪（在 LoRA 参数上）
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

## DoRA：权重分解

### 核心改进
DoRA（Weight-Decomposed Low-Rank Adaptation）将权重分解为大小（magnitude）和方向（direction）：

$$W = W_0 + \Delta W = m \frac{W_0 + BA}{||W_0 + BA||_c}$$

```python
class DoRALayer(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        
        # 冻结原始权重
        self.base_layer.weight.requires_grad = False
        
        # LoRA 矩阵
        d, k = base_layer.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, k))
        self.lora_B = nn.Parameter(torch.zeros(d, rank))
        
        # magnitude 向量
        self.magnitude = nn.Parameter(torch.ones(d))
        
        # 预计算原始权重的列向量范数
        with torch.no_grad():
            self.register_buffer('base_weight_norm', 
                               torch.norm(base_layer.weight, dim=1, keepdim=True))
    
    def forward(self, x):
        # 计算权重矩阵
        base_weight = self.base_layer.weight
        lora_weight = self.lora_B @ self.lora_A * (self.alpha / self.rank)
        
        # 组合权重
        combined_weight = base_weight + lora_weight
        
        # 归一化方向
        weight_norm = torch.norm(combined_weight, dim=1, keepdim=True)
        normalized_weight = combined_weight / (weight_norm + 1e-8)
        
        # 应用 magnitude
        final_weight = self.magnitude.unsqueeze(1) * normalized_weight
        
        return F.linear(x, final_weight, self.base_layer.bias)
```

## Adapter 系列方法

### Adapter Layers
在 Transformer 层中插入小的适配器模块：

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # 初始化接近恒等映射
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        # 残差连接确保适配器关闭时为恒等映射
        adapter_output = self.up_proj(
            self.dropout(self.activation(self.down_proj(x)))
        )
        return x + adapter_output

# 集成到 Transformer 层
class TransformerWithAdapter(nn.Module):
    def __init__(self, base_transformer, adapter_size=64):
        super().__init__()
        self.base_transformer = base_transformer
        
        # 冻结原始参数
        for param in self.base_transformer.parameters():
            param.requires_grad = False
        
        # 在每一层后添加 Adapter
        self.adapters = nn.ModuleList([
            AdapterLayer(base_transformer.config.hidden_size, adapter_size)
            for _ in range(base_transformer.config.num_hidden_layers)
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.base_transformer.layers):
            x = layer(x)
            x = self.adapters[i](x)  # Adapter 处理
        return x
```

### Prefix-Tuning
在输入序列前添加可训练的 prefix：

```python
class PrefixTuningModel(nn.Module):
    def __init__(self, base_model, prefix_length=20, prefix_dim=768):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        
        # 冻结基础模型
        for param in base_model.parameters():
            param.requires_grad = False
        
        # 可训练的 prefix 参数
        self.prefix_tokens = nn.Parameter(
            torch.randn(prefix_length, prefix_dim) * 0.02
        )
        
        # prefix 投影层（映射到键值空间）
        self.prefix_projection = nn.Sequential(
            nn.Linear(prefix_dim, 2 * base_model.config.num_attention_heads * 
                     base_model.config.hidden_size // base_model.config.num_attention_heads),
            nn.Tanh(),
            nn.Linear(2 * base_model.config.num_attention_heads * 
                     base_model.config.hidden_size // base_model.config.num_attention_heads,
                     2 * base_model.config.hidden_size)
        )
    
    def get_prefix_key_values(self, batch_size):
        """生成前缀的键值对"""
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        key_values = self.prefix_projection(prefix_tokens)
        
        # 重组为键值对格式
        key_values = key_values.view(
            batch_size, self.prefix_length, 2, 
            self.base_model.config.num_attention_heads, -1
        )
        
        return key_values.split(1, dim=2)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        
        # 获取 prefix 键值对
        prefix_keys, prefix_values = self.get_prefix_key_values(batch_size)
        
        # 扩展注意力掩码
        prefix_attention_mask = torch.ones(
            batch_size, self.prefix_length, 
            device=input_ids.device, dtype=attention_mask.dtype
        )
        extended_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        # 前向传播（需要修改模型接受外部键值对）
        return self.base_model(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            prefix_key_values=(prefix_keys, prefix_values)
        )
```

### P-Tuning v2
连续提示的改进版本，更加高效：

```python
class PTuningV2(nn.Module):
    def __init__(self, base_model, num_virtual_tokens=20):
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        
        # 虚拟 token 嵌入
        self.virtual_token_embeddings = nn.Embedding(
            num_virtual_tokens, 
            base_model.config.hidden_size
        )
        
        # 深度提示调优：每层都有独立的虚拟 token
        self.deep_prompt_embeddings = nn.Parameter(
            torch.randn(
                base_model.config.num_hidden_layers,
                num_virtual_tokens,
                base_model.config.hidden_size
            ) * 0.02
        )
    
    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        # 获取原始输入嵌入
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # 添加虚拟 token
        virtual_embeds = self.virtual_token_embeddings.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # 拼接输入
        full_embeds = torch.cat([virtual_embeds, inputs_embeds], dim=1)
        
        # 修改注意力掩码
        virtual_attention_mask = torch.ones(
            batch_size, self.num_virtual_tokens,
            device=input_ids.device
        )
        full_attention_mask = torch.cat([
            virtual_attention_mask, 
            kwargs.get('attention_mask', torch.ones_like(input_ids))
        ], dim=1)
        
        return self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask
        )
```

## 方法对比分析

### 参数量与显存对比

| 方法 | 可训练参数 | 显存占用 | 推理开销 | 训练稳定性 |
|------|------------|----------|----------|------------|
| **Full Fine-tuning** | 100% | 高 | 无 | 高 |
| **LoRA** | 0.1-1% | 低 | 极低 | 高 |
| **QLoRA** | 0.1-1% | 极低 | 极低 | 中等 |
| **DoRA** | 0.1-1% + 权重维度 | 低 | 低 | 高 |
| **Adapter** | 1-3% | 中等 | 中等 | 高 |
| **Prefix-Tuning** | 0.01-0.1% | 低 | 低 | 中等 |
| **P-Tuning v2** | 0.01-0.1% | 低 | 低 | 中等 |

### 性能对比代码
```python
def compare_peft_methods():
    """对比不同 PEFT 方法的效果"""
    base_model = "meta-llama/Llama-2-7b-hf"
    dataset = "alpaca"
    
    results = {}
    
    # LoRA 实验
    lora_config = LoraConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
    lora_results = train_and_evaluate(base_model, lora_config, dataset)
    results["LoRA"] = lora_results
    
    # QLoRA 实验
    qlora_config = LoraConfig(r=64, alpha=128, target_modules=["q_proj", "v_proj"])
    qlora_results = train_and_evaluate(base_model, qlora_config, dataset, use_quantization=True)
    results["QLoRA"] = qlora_results
    
    # Adapter 实验
    adapter_config = AdapterConfig(adapter_size=64)
    adapter_results = train_and_evaluate_adapter(base_model, adapter_config, dataset)
    results["Adapter"] = adapter_results
    
    return results

# 选型指南
class PEFTSelector:
    @staticmethod
    def recommend_method(
        model_size_gb: float,
        available_memory_gb: float, 
        target_performance: str,
        inference_latency_critical: bool
    ):
        """根据约束条件推荐 PEFT 方法"""
        
        if available_memory_gb < model_size_gb * 0.3:
            if target_performance == "high":
                return "QLoRA", "极低内存，可接受的性能损失"
            else:
                return "Prefix-Tuning", "最低内存占用"
        
        elif available_memory_gb < model_size_gb * 0.6:
            if inference_latency_critical:
                return "LoRA", "平衡内存和推理速度"
            else:
                return "DoRA", "更好的性能表现"
        
        else:
            if target_performance == "highest":
                return "Adapter", "最高性能，可接受推理开销"
            else:
                return "LoRA", "通用最佳选择"

# 使用示例
selector = PEFTSelector()
method, reason = selector.recommend_method(
    model_size_gb=13.0,     # 13B 模型
    available_memory_gb=24.0, # 24GB 显存
    target_performance="high",
    inference_latency_critical=True
)
print(f"推荐方法: {method}, 理由: {reason}")
```

## 面试常见问题

### Q1: LoRA 为什么有效？理论基础是什么？

**答案：**
1. **低秩假设**：微调过程中权重变化矩阵通常具有低秩特性，LoRA 利用这一先验
2. **内在维度理论**：神经网络的有效参数空间维度远低于实际参数数量
3. **预训练知识保持**：冻结原始权重保持预训练知识，只学习任务特定的适应
4. **实验证据**：在多个任务上展现出接近全量微调的性能

### Q2: QLoRA 的双重量化（double quantization）是什么意思？

**答案：**
1. **第一层量化**：将 FP16 权重量化为 4-bit NormalFloat
2. **第二层量化**：将量化常数（quantization constants）从 FP32 量化为 8-bit
3. **内存节省**：每个参数从 16-bit 降到 4.25-bit（4-bit + 0.25-bit 常数）
4. **精度保持**：NF4 数据类型专门为正态分布权重设计，减少量化误差

### Q3: 什么情况下应该选择 DoRA 而不是 LoRA？

**答案：**
1. **性能要求高**：DoRA 在多数任务上优于 LoRA，特别是需要改变权重大小的任务
2. **权重分析重要**：需要分析权重变化的方向和大小时
3. **可接受额外开销**：DoRA 需要额外的 magnitude 参数和范数计算
4. **具体任务**：数学推理、代码生成等需要精确权重调整的任务

### Q4: Prefix-Tuning 和 P-Tuning v2 的主要区别？

**答案：**
1. **作用位置**：Prefix-Tuning 只在输入层，P-Tuning v2 在每一层都有提示
2. **参数数量**：P-Tuning v2 参数更多但效果更好
3. **任务适应性**：P-Tuning v2 对理解类任务效果更佳
4. **实现复杂度**：Prefix-Tuning 实现更简单，P-Tuning v2 需要修改每层前向传播

### Q5: 如何为新任务选择合适的 PEFT 方法？

**答案：**
1. **资源约束**：
   - 显存紧张：QLoRA > LoRA > Adapter
   - 推理速度敏感：LoRA > DoRA > Adapter
2. **任务类型**：
   - 生成任务：LoRA、DoRA
   - 理解任务：P-Tuning v2、Prefix-Tuning
   - 多任务学习：Adapter
3. **性能要求**：
   - 最高性能：DoRA > LoRA > Adapter
   - 平衡选择：LoRA（通用性最佳）
4. **实验验证**：在验证集上对比不同方法的效果

---

---

## See Also

- [[AI/LLM/SFT/LoRA|LoRA（低秩适应）]] — PEFT家族核心成员，本文对比的基准方法；LoRA的rank选择直接决定参数效率与表达能力的折中
- [[AI/LLM/SFT/EWC-LoRA-Continual-Learning-Low-Rank|EWC-LoRA（持续学习LoRA）]] — LoRA在持续学习场景的扩展：Fisher information正则化克服灾难性遗忘，是PEFT方法在multi-task场景下的前沿演化方向
- [[AI/LLM/SFT/LLM微调实战-2026技术全景|LLM微调实战2026全景]] ⭐ — PEFT方法的工程落地指南；哪个场景用LoRA、QLoRA、DoRA，本文理论 + 微调实战提供实践配方
- [[AI/LLM/Inference/Progressive-Thought-Encoding-Cache-Efficient-RL|PTE（渐进式思维编码）]] — PEFT思想向推理阶段的迁移：PTE在KV cache被压缩时用LoRA ΔW在线蒸馏保存evicted token，本质是LoRA作为"记忆压缩器"的创新用法
- [[AI/LLM/SFT/Post-Training Unified View 论文|Post-Training 统一视角]] — PEFT是post-training工程的核心工具；统一视角论文提供了SFT/RLHF/DPO在PEFT框架下的系统性理解

> 📁 **版本说明**：`AI/LLM/SFT/PEFT-方法综述.md` 为早期简化版（343行，deprecated），本文为完整正式版（530行）。